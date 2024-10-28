// Trigger 1: Automatically create a welcome case when a new student is created
trigger StudentTrigger on Student__c (after insert) {
    List<Case> casesToInsert = new List<Case>();
    
    for(Student__c student : Trigger.new) {
        // Create a welcome case for each new student
        Case welcomeCase = new Case(
            Subject = 'Welcome to EduConsultPro',
            Description = 'Welcome to EduConsultPro! This case has been automatically created to track your initial onboarding.',
            Status = 'Open',
            Origin = 'New Student Registration',
            Priority = 'Normal',
            ContactEmail = student.Email__c,
            Student__c = student.Id  // Assuming there's a lookup field on Case to Student__c
        );
        casesToInsert.add(welcomeCase);
    }
    
    if(!casesToInsert.isEmpty()) {
        try {
            insert casesToInsert;
        } catch(DMLException e) {
            // Add error handling
            for(Student__c student : Trigger.new) {
                student.addError('Unable to create welcome case: ' + e.getMessage());
            }
        }
    }
}
