// Trigger 2: Validate appointment scheduling and send notification
trigger AppointmentTrigger on Appointment__c (before insert, before update, after insert) {
    if(Trigger.isBefore) {
        // Validate appointment scheduling
        for(Appointment__c appt : Trigger.new) {
            // Check if appointment is during business hours (9 AM to 5 PM)
            DateTime apptDateTime = DateTime.newInstance(
                appt.Appointment_Date__c, 
                appt.Appointment_Time__c
            );
            
            Integer hourOfDay = apptDateTime.hour();
            
            if(hourOfDay < 9 || hourOfDay >= 17) {
                appt.addError('Appointments can only be scheduled between 9 AM and 5 PM');
            }
            
            // Check if consultant is available at the requested time
            if(Trigger.isInsert || (Trigger.isUpdate && 
                appt.Consultant__c != Trigger.oldMap.get(appt.Id).Consultant__c ||
                appt.Appointment_Date__c != Trigger.oldMap.get(appt.Id).Appointment_Date__c ||
                appt.Appointment_Time__c != Trigger.oldMap.get(appt.Id).Appointment_Time__c)) {
                
                // Query for existing appointments
                List<Appointment__c> existingAppointments = [
                    SELECT Id 
                    FROM Appointment__c 
                    WHERE Consultant__c = :appt.Consultant__c 
                    AND Appointment_Date__c = :appt.Appointment_Date__c
                    AND Appointment_Time__c = :appt.Appointment_Time__c
                    AND Id != :appt.Id
                ];
                
                if(!existingAppointments.isEmpty()) {
                    appt.addError('Consultant is already booked for this time slot');
                }
            }
        }
    }
    
    if(Trigger.isAfter && Trigger.isInsert) {
        // Send email notifications
        List<Messaging.SingleEmailMessage> emailsToSend = new List<Messaging.SingleEmailMessage>();
        
        for(Appointment__c appt : Trigger.new) {
            // Query for related records to get email addresses
            Appointment__c apptWithRelations = [
                SELECT Id, Student__r.Email__c, Consultant__r.Email__c,
                       Appointment_Date__c, Appointment_Time__c
                FROM Appointment__c 
                WHERE Id = :appt.Id
            ];
            
            // Create email for student
            if(apptWithRelations.Student__r.Email__c != null) {
                Messaging.SingleEmailMessage studentEmail = new Messaging.SingleEmailMessage();
                studentEmail.setToAddresses(new List<String>{apptWithRelations.Student__r.Email__c});
                studentEmail.setSubject('Appointment Confirmation');
                studentEmail.setPlainTextBody(
                    'Your appointment has been scheduled for ' + 
                    apptWithRelations.Appointment_Date__c.format() + ' at ' + 
                    apptWithRelations.Appointment_Time__c.format() + '.'
                );
                emailsToSend.add(studentEmail);
            }
            
            // Create email for consultant
            if(apptWithRelations.Consultant__r.Email__c != null) {
                Messaging.SingleEmailMessage consultantEmail = new Messaging.SingleEmailMessage();
                consultantEmail.setToAddresses(new List<String>{apptWithRelations.Consultant__r.Email__c});
                consultantEmail.setSubject('New Appointment Scheduled');
                consultantEmail.setPlainTextBody(
                    'A new appointment has been scheduled for ' + 
                    apptWithRelations.Appointment_Date__c.format() + ' at ' + 
                    apptWithRelations.Appointment_Time__c.format() + '.'
                );
                emailsToSend.add(consultantEmail);
            }
        }
        
        if(!emailsToSend.isEmpty()) {
            try {
                Messaging.sendEmail(emailsToSend);
            } catch(Exception e) {
                // Handle email sending errors
                System.debug('Error sending appointment notification emails: ' + e.getMessage());
            }
        }
    }
}
